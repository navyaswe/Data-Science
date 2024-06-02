fitted_pipeline = titanic_transformer.fit(X_train, y_train)  # just fit method called
import joblib
joblib.dump(fitted_pipeline, 'fitted_pipeline.pkl')
